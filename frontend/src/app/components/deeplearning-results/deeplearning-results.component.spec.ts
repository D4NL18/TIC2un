import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DeeplearningResultsComponent } from './deeplearning-results.component';

describe('DeeplearningResultsComponent', () => {
  let component: DeeplearningResultsComponent;
  let fixture: ComponentFixture<DeeplearningResultsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DeeplearningResultsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DeeplearningResultsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
