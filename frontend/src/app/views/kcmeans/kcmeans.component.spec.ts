import { ComponentFixture, TestBed } from '@angular/core/testing';

import { KcmeansComponent } from './kcmeans.component';

describe('KcmeansComponent', () => {
  let component: KcmeansComponent;
  let fixture: ComponentFixture<KcmeansComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [KcmeansComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(KcmeansComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
