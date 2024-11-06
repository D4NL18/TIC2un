import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SomComponent } from './som.component';

describe('SomComponent', () => {
  let component: SomComponent;
  let fixture: ComponentFixture<SomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SomComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
